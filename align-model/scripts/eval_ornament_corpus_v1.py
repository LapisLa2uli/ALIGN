"""Frozen external evaluation of Track B and grammar mapper v2 on ORN.

The three phases are intentionally separate:

* ``audit`` reads corpus metadata/targets and freezes a deterministic manifest.
* ``infer`` reads only manifest-declared performance audio and frozen artifacts.
* ``evaluate`` verifies the prediction freeze before opening canonical targets.

No ORN labels, predictions, or scores participate in model/config selection.
"""

from __future__ import annotations

import argparse
import atexit
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from alignmodel.joint.baseline import current_note_aligner_baseline
from alignmodel.joint.grammar_mapper_v2 import GrammarCosts, decode_grammar_mapper
from alignmodel.joint.index import JointEvent, ScoreEventIndex
from alignmodel.joint.lattice import JointCandidate
from alignmodel.joint.metrics import (
    JointMetricSample,
    evaluate_joint_dataset,
    evaluate_joint_events,
    pair_exact_pitch_onset,
)
from alignmodel.joint.packed_data import sha256_file
from alignmodel.training_resources import resource_lease
from alignmodel.transcription.mel_v1 import (
    decode_mel_notes,
    extract_log_mel,
    infer_mel_probabilities,
    load_audio_mono,
    load_mel_checkpoint,
)


SCHEMA_VERSION = "align-orn-external-eval-v1"
DEFAULT_SEED = 20260917
DEFAULT_LIMIT = 500
EXPECTED_CHECKPOINT_SHA256 = (
    "3d8f93732a810c3f8470a88316debb9f92b4680b2333c2187866e6090a730a4e"
)
DEFAULT_CHECKPOINT = Path(
    "runs/joint-outputraw-full-v1/mel-transcriber-v1/"
    "full-training-all4544-v2/candidate-epoch-018.pt"
)
DEFAULT_MAPPER_REPORT = Path(
    "runs/joint-outputraw-full-v1/mel-mapper-v2/full358-mel-grammar-v1.json"
)
DEFAULT_MAPPER_STATUS = Path(
    "runs/joint-outputraw-full-v1/mel-mapper-v2/MAPPER_V2_STATUS.json"
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
HASHED_SELECTED_FILES = REQUIRED_FILES
RANDOM_SOURCE_NAMES = frozenset({"", "gen", "generated", "procedural", "random"})


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _stable(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


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


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
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
                    compressed.write(
                        json.dumps(
                            row, sort_keys=True, ensure_ascii=False
                        ).encode("utf-8")
                    )
                    compressed.write(b"\n")
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_audit_module(repo: Path) -> Any:
    path = repo / "scripts" / "audit_training_data.py"
    spec = importlib.util.spec_from_file_location("orn_frozen_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load corpus audit module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RecordingIssueLog:
    """IssueLog-compatible recorder retaining exclusion reasons per bundle."""

    def __init__(self) -> None:
        self.counts: Counter[tuple[str, str]] = Counter()
        self.examples: defaultdict[tuple[str, str], list[dict[str, Any]]] = (
            defaultdict(list)
        )
        self.by_sample: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    def add(
        self,
        code: str,
        severity: str,
        sample: str,
        detail: str,
        *,
        path: str | None = None,
    ) -> None:
        key = (str(code), str(severity))
        row = {"code": key[0], "severity": key[1], "detail": str(detail)}
        if path is not None:
            row["path"] = str(path)
        self.counts[key] += 1
        if len(self.examples[key]) < 12:
            self.examples[key].append({"sample": str(sample), **row})
        self.by_sample[str(sample)].append(row)

    def report(self) -> list[dict[str, Any]]:
        rank = {"critical": 0, "high": 1, "warning": 2, "info": 3}
        return sorted(
            (
                {
                    "code": code,
                    "severity": severity,
                    "count": count,
                    "examples": self.examples[(code, severity)],
                }
                for (code, severity), count in self.counts.items()
            ),
            key=lambda row: (
                rank.get(str(row["severity"]), 9),
                -int(row["count"]),
                str(row["code"]),
            ),
        )


def _source_identity(
    metadata: Mapping[str, Any], verified_score_sha256: str
) -> tuple[str, str, str]:
    declared = str(metadata.get("source") or "").strip()
    source_score = str(metadata.get("source_score") or "").strip()
    if declared.casefold() in RANDOM_SOURCE_NAMES:
        return (
            "random_generated",
            f"generated:{verified_score_sha256}",
            "random-generated",
        )
    stem = Path(source_score or declared).stem.casefold()
    if not stem:
        raise ValueError("Raw-score bundle has no authoritative source identifier")
    return "raw_score", f"score:{stem}", f"raw-score:{stem}"


def _max_polyphony(rows: Sequence[Mapping[str, Any]]) -> int:
    points = []
    for row in rows:
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        if end - start <= 0.005:
            continue
        # Ends sort before starts at the same instant.
        points.extend(((start, 1), (end, -1)))
    active = maximum = 0
    for _time, change in sorted(points, key=lambda item: (item[0], item[1])):
        active += change
        maximum = max(maximum, active)
    return maximum


def _ornament_stats(lineage: Mapping[str, Any]) -> dict[str, Any]:
    rendered = list(lineage.get("rendered_notes") or [])
    unmapped = [
        row for row in rendered if not (row.get("performed_indices") or [])
    ]
    return {
        "rendered_events": len(rendered),
        "performed_events": len(lineage.get("performed_notes") or []),
        "unmapped_rendered_events": len(unmapped),
        "unmapped_rendered_fraction": len(unmapped) / max(len(rendered), 1),
        "max_simultaneous_rendered_events": _max_polyphony(rendered),
        "outside_track_b_written_pitch_range": sum(
            not 52 <= int(row.get("pitch_midi_written", -999)) <= 100
            for row in rendered
        ),
    }


def _custom_audit_reasons(
    sample_dir: Path,
    metadata: Mapping[str, Any],
    labels: Mapping[str, Any],
    lineage: Mapping[str, Any] | None,
    record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reasons = []

    def reject(code: str, detail: str) -> None:
        reasons.append({"code": code, "severity": "critical", "detail": detail})

    missing = [name for name in REQUIRED_FILES if not (sample_dir / name).is_file()]
    if missing:
        reject("orn_required_file_missing", ", ".join(missing))
    if str(metadata.get("schema_version")) != "1.2":
        reject("orn_metadata_schema_invalid", repr(metadata.get("schema_version")))
    if str(labels.get("schema_version")) != "1.2":
        reject("orn_labels_schema_invalid", repr(labels.get("schema_version")))
    if lineage is None or str(lineage.get("schema_version")) != "1.0":
        reject(
            "orn_note_map_schema_invalid",
            repr(None if lineage is None else lineage.get("schema_version")),
        )
    if int(record.get("invalid_label_count") or 0):
        reject(
            "orn_invalid_label_rows",
            str(record.get("invalid_label_count")),
        )
    expected_pitch = {
        "audio_pitch_space": "sounding",
        "midi_pitch_space": "sounding",
        "effective_audio_transpose": 2,
        "sounding_transpose": -2,
    }
    actual_pitch = {
        key: record.get(key)
        for key in (
            "audio_pitch_space",
            "midi_pitch_space",
            "effective_audio_transpose",
            "sounding_transpose",
        )
    }
    if actual_pitch != expected_pitch:
        reject(
            "orn_pitch_convention_invalid",
            f"expected={expected_pitch} actual={actual_pitch}",
        )
    declared = str(metadata.get("source") or "").casefold()
    if declared not in RANDOM_SOURCE_NAMES:
        source_score = Path(str(metadata.get("source_score") or ""))
        if not source_score.is_file():
            reject("orn_raw_source_score_missing", str(source_score))
    return reasons


def _equal_quotas(
    capacities: Mapping[str, int], total: int, seed: int
) -> dict[str, int]:
    """Round-robin equal allocation with deterministic capacity redistribution."""

    quotas = {key: 0 for key in capacities}
    order = sorted(capacities, key=lambda key: (_stable(seed, key), key))
    remaining = min(int(total), sum(max(0, int(value)) for value in capacities.values()))
    while remaining:
        progressed = False
        for key in order:
            if remaining == 0:
                break
            if quotas[key] >= int(capacities[key]):
                continue
            quotas[key] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break
    return quotas


def _select_grouped(
    records: Sequence[Mapping[str, Any]],
    *,
    group_key: str,
    total: int,
    seed: int,
) -> list[dict[str, Any]]:
    groups: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row[group_key])].append(row)
    quotas = _equal_quotas(
        {key: len(values) for key, values in groups.items()}, total, seed
    )
    selected = []
    for key, values in groups.items():
        ordered = sorted(
            values,
            key=lambda row: (
                _stable(seed + 1, str(row["sample"])),
                str(row["sample"]),
            ),
        )
        selected.extend(dict(row) for row in ordered[: quotas[key]])
    return selected


def select_balanced_manifest_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Balance raw/random provenance, then raw works/generated score identities."""

    eligible = [row for row in records if bool(row.get("eligible"))]
    provenance = defaultdict(list)
    for row in eligible:
        provenance[str(row["provenance"])].append(row)
    provenance_quotas = _equal_quotas(
        {key: len(values) for key, values in provenance.items()},
        min(int(limit), len(eligible)),
        seed,
    )
    selected = []
    for key, values in provenance.items():
        selected.extend(
            _select_grouped(
                values,
                group_key="source_song_id",
                total=provenance_quotas[key],
                seed=seed + int(_stable(seed, key)[:8], 16),
            )
        )
    return sorted(
        selected,
        key=lambda row: (
            _stable(seed + 2, str(row["sample"])),
            str(row["sample"]),
        ),
    )


def exclude_duplicate_candidates(
    records: Sequence[dict[str, Any]], *, seed: int
) -> dict[str, int]:
    """Retain one deterministic representative per content and audio identity."""

    counts = {"content": 0, "performance_audio": 0}
    for field, hash_name, reason in (
        ("content_fingerprint", None, "duplicate_bundle_content"),
        (None, "performance_audio.wav", "duplicate_performance_audio"),
    ):
        seen: dict[str, str] = {}
        for row in sorted(
            records,
            key=lambda value: (
                _stable(seed, str(value["sample"])),
                str(value["sample"]),
            ),
        ):
            if not row["eligible"]:
                continue
            identity = (
                str(row.get(field) or "")
                if field is not None
                else str((row.get("hashes") or {}).get(hash_name) or "")
            )
            previous = seen.setdefault(identity, str(row["sample"]))
            if identity and previous != row["sample"]:
                row["eligible"] = False
                row["exclusion_reasons"].append(
                    {
                        "code": reason,
                        "severity": "critical",
                        "detail": f"same identity as {previous}",
                    }
                )
                counts[
                    "content" if field is not None else "performance_audio"
                ] += 1
    return counts


def _public_inventory_row(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if not str(key).startswith("_")
    }


def _audit_one(
    audit: Any,
    source_root: Path,
    sample_dir: Path,
    hash_cache: dict[str, dict[str, Any]],
    issues: RecordingIssueLog,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
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
    metadata = _load_json(sample_dir / "metadata.json")
    labels = _load_json(sample_dir / "labels.json")
    lineage = result.get("_note_map_document")
    reasons = [
        row
        for row in issues.by_sample.get(sample_dir.name, [])
        if str(row.get("severity")) == "critical"
    ]
    reasons.extend(
        _custom_audit_reasons(
            sample_dir,
            metadata,
            labels,
            lineage if isinstance(lineage, Mapping) else None,
            result,
        )
    )
    verified_hash = str((result.get("hashes") or {}).get("verified_score.musicxml") or "")
    try:
        provenance, source_song_id, sampling_stratum = _source_identity(
            metadata, verified_hash
        )
    except ValueError as exc:
        provenance, source_song_id, sampling_stratum = (
            "unknown",
            f"invalid:{sample_dir.name}",
            "invalid",
        )
        reasons.append(
            {
                "code": "orn_source_identity_invalid",
                "severity": "critical",
                "detail": str(exc),
            }
        )
    source_score = str(metadata.get("source_score") or "")
    source_score_hash = (
        audit._sha256(Path(source_score), hash_cache)
        if source_score and Path(source_score).is_file()
        else None
    )
    row = {
        "sample": sample_dir.name,
        "sample_dir": str(sample_dir.resolve()),
        "provenance": provenance,
        "source_song_id": source_song_id,
        "sampling_stratum": sampling_stratum,
        "declared_source": str(metadata.get("source") or ""),
        "source_score": source_score or None,
        "source_score_sha256": source_score_hash,
        "paths": {
            name: str((sample_dir / name).resolve())
            for name in REQUIRED_FILES
            if (sample_dir / name).is_file()
        },
        "hashes": dict(result.get("hashes") or {}),
        "schemas": {
            "metadata": metadata.get("schema_version"),
            "labels": labels.get("schema_version"),
            "note_map": (
                lineage.get("schema_version")
                if isinstance(lineage, Mapping)
                else None
            ),
        },
        "duration_sec": result.get("duration_sec"),
        "audio_pitch_space": result.get("audio_pitch_space"),
        "midi_pitch_space": result.get("midi_pitch_space"),
        "effective_audio_transpose": result.get("effective_audio_transpose"),
        "sounding_transpose": result.get("sounding_transpose"),
        "note_map_valid": bool(result.get("note_map_valid")),
        "note_map_stats": result.get("note_map_stats") or {},
        "ornament_realization": (
            _ornament_stats(lineage)
            if isinstance(lineage, Mapping)
            else None
        ),
        "content_fingerprint": result.get("content_fingerprint"),
        "target_lineage_sha256": (
            hashlib.sha256(_canonical_bytes(lineage)).hexdigest()
            if isinstance(lineage, Mapping)
            else None
        ),
        "eligible": bool(result.get("eligible")) and not reasons,
        "exclusion_reasons": reasons,
        "selected": False,
        "canonical_projection_checked": False,
    }
    return row, dict(lineage) if isinstance(lineage, Mapping) else None


def _known_training_sources(final_status: Mapping[str, Any]) -> list[str]:
    keys = (
        final_status.get("score_agnostic_diagnostics", {})
        .get("per_source", {})
        .keys()
    )
    return sorted(f"score:{str(value).casefold()}" for value in keys)


def run_audit(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Frozen manifest already exists: {manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    bundles = sorted(
        (path for path in source_root.iterdir() if path.is_dir()),
        key=lambda path: os.path.normcase(str(path)),
    )
    if not bundles:
        raise ValueError(f"No ORN bundles found: {source_root}")
    audit = _load_audit_module(repo)
    issues = RecordingIssueLog()
    hash_cache_path = output_dir / "audit_hash_cache.json"
    hash_cache = (
        _load_json(hash_cache_path) if hash_cache_path.is_file() else {}
    )
    records = []
    lease = resource_lease(
        args.resource_status,
        "cpu_validation",
        track="orn-eval-500-v1-audit",
        command=[sys.executable, *sys.argv],
        metadata={"source_root": str(source_root), "locked_test": False},
    )
    lease.__enter__()
    atexit.register(lease.__exit__, None, None, None)
    try:
        for position, sample_dir in enumerate(bundles, 1):
            row, _lineage = _audit_one(
                audit, source_root, sample_dir, hash_cache, issues
            )
            records.append(row)
            if position % 250 == 0 or position == len(bundles):
                _atomic_json(hash_cache_path, hash_cache)
                print(f"audit={position}/{len(bundles)}", flush=True)

        # Exact duplicate bundles/audio never compete for sampling quota.
        duplicate_exclusions = exclude_duplicate_candidates(
            records, seed=args.seed
        )

        # Projection is expensive and required only for rows that can enter the
        # frozen sample. Deterministic refill continues until every selected row
        # has a valid canonical target or all audit-eligible rows are exhausted.
        selected_lineages: dict[str, dict[str, Any]] = {}
        while True:
            selected = select_balanced_manifest_rows(
                records, limit=args.limit, seed=args.seed
            )
            failures = 0
            for row in selected:
                stored = next(
                    value for value in records if value["sample"] == row["sample"]
                )
                if stored["canonical_projection_checked"]:
                    continue
                local_issues = RecordingIssueLog()
                checked, lineage = _audit_one(
                    audit,
                    source_root,
                    Path(str(stored["sample_dir"])),
                    hash_cache,
                    local_issues,
                )
                if lineage is None:
                    error = "audited note-map target is unavailable"
                else:
                    try:
                        index = ScoreEventIndex.from_musicxml(
                            Path(str(stored["sample_dir"]))
                            / "verified_score.musicxml",
                            lineage,
                        )
                        if not index.rendered_events:
                            raise ValueError("no rendered target events")
                        error = None
                    except Exception as exc:  # fail closed per selected row
                        error = f"{type(exc).__name__}: {exc}"
                stored["canonical_projection_checked"] = True
                if error is not None:
                    stored["eligible"] = False
                    stored["exclusion_reasons"].append(
                        {
                            "code": "canonical_projection_invalid",
                            "severity": "critical",
                            "detail": error,
                        }
                    )
                    failures += 1
                else:
                    stored.update(
                        {
                            key: checked[key]
                            for key in (
                                "target_lineage_sha256",
                                "ornament_realization",
                                "note_map_stats",
                            )
                        }
                    )
                    selected_lineages[str(stored["sample"])] = lineage
            if failures == 0:
                break

        selected = select_balanced_manifest_rows(
            records, limit=args.limit, seed=args.seed
        )
        if len(selected) != min(
            args.limit, sum(bool(row["eligible"]) for row in records)
        ):
            raise RuntimeError("Balanced selection did not fill available quota")
        selected_names = {str(row["sample"]) for row in selected}
        by_sample = {str(row["sample"]): row for row in records}
        for sample in selected_names:
            by_sample[sample]["selected"] = True
            sample_dir = Path(str(by_sample[sample]["sample_dir"]))
            by_sample[sample]["hashes"] = {
                name: audit._sha256(sample_dir / name, hash_cache)
                for name in HASHED_SELECTED_FILES
            }
            if sample not in selected_lineages:
                _checked, lineage = _audit_one(
                    audit, source_root, sample_dir, hash_cache, RecordingIssueLog()
                )
                if lineage is None:
                    raise RuntimeError(f"Selected target disappeared: {sample}")
                selected_lineages[sample] = lineage
        selected = [dict(by_sample[str(row["sample"])]) for row in selected]

        target_path = output_dir / "canonical_targets.jsonl.gz"
        _atomic_jsonl_gz(
            target_path,
            (
                {
                    "sample": row["sample"],
                    "target_lineage_sha256": row["target_lineage_sha256"],
                    "lineage": selected_lineages[str(row["sample"])],
                }
                for row in selected
            ),
        )
        inventory_path = output_dir / "inventory.jsonl.gz"
        _atomic_jsonl_gz(
            inventory_path,
            (_public_inventory_row(row) for row in records),
        )
        _atomic_json(hash_cache_path, hash_cache)

        song_available = Counter(
            str(row["source_song_id"]) for row in records if row["eligible"]
        )
        song_selected = Counter(str(row["source_song_id"]) for row in selected)
        provenance_available = Counter(
            str(row["provenance"]) for row in records if row["eligible"]
        )
        provenance_selected = Counter(str(row["provenance"]) for row in selected)
        fingerprint_counts = Counter(
            str(row["content_fingerprint"]) for row in selected
        )
        audio_counts = Counter(
            str(row["hashes"]["performance_audio.wav"]) for row in selected
        )
        checkpoint = args.checkpoint.resolve()
        mapper_report = args.mapper_report.resolve()
        mapper_status = args.mapper_status.resolve()
        final_status_path = checkpoint.parents[1] / "FINAL_STATUS.json"
        final_status = _load_json(final_status_path)
        if sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
            raise ValueError("Frozen Track B checkpoint SHA-256 mismatch")
        mapper_reference = _load_json(mapper_report)
        mapper_state = _load_json(mapper_status)
        if mapper_reference.get("schema_version") != (
            "align-grammar-mapper-v2-calibration-v1"
        ):
            raise ValueError("Reference mapper is not validated grammar mapper v2")
        if mapper_reference.get("costs") != GrammarCosts().__dict__:
            raise ValueError("Reference mapper costs differ from frozen grammar v1")
        if (
            mapper_state.get("full_358_combined", {}).get("report_sha256")
            != sha256_file(mapper_report)
        ):
            raise ValueError("Mapper status does not attest the reference report")
        known_training = _known_training_sources(final_status)
        selected_sources = {str(row["source_song_id"]) for row in selected}
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "phase": "frozen_manifest",
            "created_utc": _utc(),
            "command": [sys.executable, *sys.argv],
            "source_root": str(source_root),
            "seed": args.seed,
            "requested_rows": args.limit,
            "selection_policy": {
                "outcome_blind": True,
                "fields_used": [
                    "audit eligibility",
                    "raw/random provenance",
                    "authoritative source-song identity",
                    "seeded sample identity",
                ],
                "fields_not_used": [
                    "labels",
                    "error type",
                    "predictions",
                    "difficulty",
                    "ORN outcomes",
                ],
                "allocation": (
                    "equal deterministic round-robin across raw/random provenance; "
                    "then equal round-robin across raw source-score stems or unique "
                    "generated verified-score hashes; exhausted capacity is "
                    "redistributed deterministically"
                ),
            },
            "counts": {
                "bundles_discovered": len(records),
                "audit_eligible": sum(bool(row["eligible"]) for row in records),
                "excluded": sum(not bool(row["eligible"]) for row in records),
                "selected": len(selected),
                "distinct_available_song_ids": len(song_available),
                "distinct_selected_song_ids": len(song_selected),
                "provenance_available": dict(sorted(provenance_available.items())),
                "provenance_selected": dict(sorted(provenance_selected.items())),
            },
            "per_song": {
                key: {
                    "provenance": (
                        "random_generated"
                        if key.startswith("generated:")
                        else "raw_score"
                    ),
                    "available": song_available[key],
                    "selected": song_selected[key],
                }
                for key in sorted(song_available)
            },
            "artifacts": {
                "inventory": {
                    "path": str(inventory_path),
                    "sha256": sha256_file(inventory_path),
                },
                "canonical_targets": {
                    "path": str(target_path),
                    "sha256": sha256_file(target_path),
                },
                "audit_hash_cache": {
                    "path": str(hash_cache_path),
                    "sha256": sha256_file(hash_cache_path),
                },
            },
            "models": {
                "transcriber_checkpoint": {
                    "path": str(checkpoint),
                    "sha256": sha256_file(checkpoint),
                    "min_confidence": 0.8,
                    "final_status": str(final_status_path),
                    "final_status_sha256": sha256_file(final_status_path),
                },
                "mapper": {
                    "name": "mel-mapper-v2 grammar-v1",
                    "reference_report": str(mapper_report),
                    "reference_report_sha256": sha256_file(mapper_report),
                    "status": str(mapper_status),
                    "status_sha256": sha256_file(mapper_status),
                    "costs": GrammarCosts().__dict__,
                },
            },
            "selected_rows": selected,
            "audit": {
                "issues": issues.report(),
                "duplicate_exclusions": duplicate_exclusions,
                "pitch_policy": (
                    "Bb clarinet written targets from sounding audio; "
                    "sounding=-2 and learned output mapping=+2"
                ),
                "official_targets": (
                    "selected rows passed ScoreEventIndex tie-aware MusicXML "
                    "projection; rendered extras retain exclusive rendered_index"
                ),
            },
            "duplicate_and_leakage_proof": {
                "unique_sample_paths": len(
                    {os.path.normcase(str(row["sample_dir"])) for row in selected}
                )
                == len(selected),
                "unique_bundle_content": len(fingerprint_counts) == len(selected),
                "unique_performance_audio": len(audio_counts) == len(selected),
                "duplicate_content_count": sum(
                    count - 1 for count in fingerprint_counts.values()
                ),
                "duplicate_audio_count": sum(
                    count - 1 for count in audio_counts.values()
                ),
                "known_training_source_song_ids_from_final_status": known_training,
                "selected_known_training_source_overlap": sorted(
                    selected_sources & set(known_training)
                ),
                "note": (
                    "Multiple bundles per raw source song are intentional quota "
                    "members, not duplicate bundles. Mozart source-song overlap "
                    "is disclosed rather than treated as source-disjoint."
                ),
            },
            "isolation": {
                "locked_test_accessed": False,
                "production_weights_mutated": False,
                "ORN_used_for_tuning": False,
            },
        }
        _atomic_json(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "manifest": str(manifest_path),
                    "selected": len(selected),
                    "provenance": dict(provenance_selected),
                },
                indent=2,
            ),
            flush=True,
        )
    finally:
        atexit.unregister(lease.__exit__)
        lease.__exit__(None, None, None)


def run_refreeze(args: argparse.Namespace) -> None:
    """Repair a pre-inference draft using its complete audited inventory."""

    repo = args.repo.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    predictions_path = output_dir / "predictions.jsonl"
    if predictions_path.exists() or (output_dir / "freeze_manifest.json").exists():
        raise FileExistsError("Cannot refreeze after prediction inference")
    prior_manifest_sha256 = sha256_file(manifest_path)
    manifest = _load_json(manifest_path)
    inventory_path = Path(str(manifest["artifacts"]["inventory"]["path"]))
    targets_path = Path(str(manifest["artifacts"]["canonical_targets"]["path"]))
    if sha256_file(inventory_path) != manifest["artifacts"]["inventory"]["sha256"]:
        raise ValueError("Prior inventory checksum mismatch")
    if sha256_file(targets_path) != manifest["artifacts"]["canonical_targets"]["sha256"]:
        raise ValueError("Prior canonical target checksum mismatch")
    with gzip.open(inventory_path, "rt", encoding="utf-8") as stream:
        records = [
            json.loads(line)
            for line in stream
            if line.strip()
        ]
    existing_targets = _read_targets(targets_path)
    duplicate_exclusions = exclude_duplicate_candidates(
        records, seed=args.seed
    )
    audit = _load_audit_module(repo)
    hash_cache_path = Path(
        str(manifest["artifacts"]["audit_hash_cache"]["path"])
    )
    hash_cache = _load_json(hash_cache_path)
    source_root = Path(str(manifest["source_root"]))
    lease = resource_lease(
        args.resource_status,
        "cpu_support",
        track="orn-eval-500-v1-refreeze",
        command=[sys.executable, *sys.argv],
        metadata={"locked_test": False, "prediction_inference_started": False},
    )
    lease.__enter__()
    atexit.register(lease.__exit__, None, None, None)
    try:
        lineages = {
            sample: dict(row["lineage"])
            for sample, row in existing_targets.items()
        }
        by_sample = {str(row["sample"]): row for row in records}
        while True:
            selected = select_balanced_manifest_rows(
                records, limit=args.limit, seed=args.seed
            )
            failures = 0
            for row in selected:
                sample = str(row["sample"])
                stored = by_sample[sample]
                lineage = lineages.get(sample)
                if (
                    lineage is not None
                    and bool(stored.get("canonical_projection_checked"))
                ):
                    continue
                if lineage is None:
                    checked, lineage = _audit_one(
                        audit,
                        source_root,
                        Path(str(stored["sample_dir"])),
                        hash_cache,
                        RecordingIssueLog(),
                    )
                    if lineage is not None:
                        stored.update(
                            {
                                key: checked[key]
                                for key in (
                                    "target_lineage_sha256",
                                    "ornament_realization",
                                    "note_map_stats",
                                )
                            }
                        )
                try:
                    if lineage is None:
                        raise ValueError("audited note-map target is unavailable")
                    index = ScoreEventIndex.from_musicxml(
                        Path(str(stored["paths"]["verified_score.musicxml"])),
                        lineage,
                    )
                    if not index.rendered_events:
                        raise ValueError("no rendered target events")
                except Exception as exc:
                    stored["eligible"] = False
                    stored["canonical_projection_checked"] = True
                    stored["exclusion_reasons"].append(
                        {
                            "code": "canonical_projection_invalid",
                            "severity": "critical",
                            "detail": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    failures += 1
                    continue
                stored["canonical_projection_checked"] = True
                lineages[sample] = lineage
            if failures == 0:
                break

        selected = select_balanced_manifest_rows(
            records, limit=args.limit, seed=args.seed
        )
        selected_names = {str(row["sample"]) for row in selected}
        for row in records:
            row["selected"] = str(row["sample"]) in selected_names
        for sample in selected_names:
            row = by_sample[sample]
            sample_dir = Path(str(row["sample_dir"]))
            row["hashes"] = {
                name: audit._sha256(sample_dir / name, hash_cache)
                for name in HASHED_SELECTED_FILES
            }
            row["target_lineage_sha256"] = hashlib.sha256(
                _canonical_bytes(lineages[sample])
            ).hexdigest()
        selected = [dict(by_sample[str(row["sample"])]) for row in selected]
        _atomic_jsonl_gz(
            targets_path,
            (
                {
                    "sample": row["sample"],
                    "target_lineage_sha256": row["target_lineage_sha256"],
                    "lineage": lineages[str(row["sample"])],
                }
                for row in selected
            ),
        )
        _atomic_jsonl_gz(
            inventory_path,
            (_public_inventory_row(row) for row in records),
        )
        _atomic_json(hash_cache_path, hash_cache)

        song_available = Counter(
            str(row["source_song_id"]) for row in records if row["eligible"]
        )
        song_selected = Counter(str(row["source_song_id"]) for row in selected)
        provenance_available = Counter(
            str(row["provenance"]) for row in records if row["eligible"]
        )
        provenance_selected = Counter(str(row["provenance"]) for row in selected)
        fingerprint_counts = Counter(
            str(row["content_fingerprint"]) for row in selected
        )
        audio_counts = Counter(
            str(row["hashes"]["performance_audio.wav"]) for row in selected
        )
        manifest["created_utc"] = _utc()
        manifest["command"] = [sys.executable, *sys.argv]
        manifest["refrozen_from_manifest_sha256"] = prior_manifest_sha256
        manifest["selection_policy"]["duplicate_policy"] = (
            "one deterministic representative per exact bundle-content and "
            "performance-audio SHA-256 before quota allocation"
        )
        manifest["counts"] = {
            "bundles_discovered": len(records),
            "audit_eligible": sum(bool(row["eligible"]) for row in records),
            "excluded": sum(not bool(row["eligible"]) for row in records),
            "selected": len(selected),
            "distinct_available_song_ids": len(song_available),
            "distinct_selected_song_ids": len(song_selected),
            "provenance_available": dict(sorted(provenance_available.items())),
            "provenance_selected": dict(sorted(provenance_selected.items())),
        }
        manifest["per_song"] = {
            key: {
                "provenance": (
                    "random_generated"
                    if key.startswith("generated:")
                    else "raw_score"
                ),
                "available": song_available[key],
                "selected": song_selected[key],
            }
            for key in sorted(song_available)
        }
        manifest["selected_rows"] = selected
        manifest["artifacts"]["inventory"]["sha256"] = sha256_file(inventory_path)
        manifest["artifacts"]["canonical_targets"]["sha256"] = sha256_file(
            targets_path
        )
        manifest["artifacts"]["audit_hash_cache"]["sha256"] = sha256_file(
            hash_cache_path
        )
        manifest["audit"]["duplicate_exclusions"] = duplicate_exclusions
        selected_sources = {str(row["source_song_id"]) for row in selected}
        known_training = set(
            manifest["duplicate_and_leakage_proof"][
                "known_training_source_song_ids_from_final_status"
            ]
        )
        manifest["duplicate_and_leakage_proof"].update(
            {
                "unique_sample_paths": len(
                    {
                        os.path.normcase(str(row["sample_dir"]))
                        for row in selected
                    }
                )
                == len(selected),
                "unique_bundle_content": len(fingerprint_counts) == len(selected),
                "unique_performance_audio": len(audio_counts) == len(selected),
                "duplicate_content_count": sum(
                    count - 1 for count in fingerprint_counts.values()
                ),
                "duplicate_audio_count": sum(
                    count - 1 for count in audio_counts.values()
                ),
                "selected_known_training_source_overlap": sorted(
                    selected_sources & known_training
                ),
            }
        )
        if not all(
            (
                manifest["duplicate_and_leakage_proof"]["unique_sample_paths"],
                manifest["duplicate_and_leakage_proof"]["unique_bundle_content"],
                manifest["duplicate_and_leakage_proof"][
                    "unique_performance_audio"
                ],
            )
        ):
            raise RuntimeError("Refrozen selection still contains duplicates")
        _atomic_json(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "manifest": str(manifest_path),
                    "selected": len(selected),
                    "duplicate_exclusions": duplicate_exclusions,
                    "provenance": dict(provenance_selected),
                },
                indent=2,
            ),
            flush=True,
        )
    finally:
        atexit.unregister(lease.__exit__)
        lease.__exit__(None, None, None)


def _cache_key(
    row: Mapping[str, Any],
    checkpoint_sha256: str,
    frontend: Mapping[str, Any],
    decode: Mapping[str, Any],
    *,
    window_frames: int,
    overlap_frames: int,
) -> str:
    return hashlib.sha256(
        _canonical_bytes(
            {
                "schema": f"{SCHEMA_VERSION}-transcriber-cache",
                "audio_sha256": row["hashes"]["performance_audio.wav"],
                "checkpoint_sha256": checkpoint_sha256,
                "frontend": frontend,
                "decode": decode,
                "window_frames": window_frames,
                "overlap_frames": overlap_frames,
            }
        )
    ).hexdigest()


def run_infer(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported ORN manifest")
    checkpoint = Path(
        str(manifest["models"]["transcriber_checkpoint"]["path"])
    )
    checkpoint_sha = sha256_file(checkpoint)
    if (
        checkpoint_sha != EXPECTED_CHECKPOINT_SHA256
        or checkpoint_sha
        != manifest["models"]["transcriber_checkpoint"]["sha256"]
    ):
        raise ValueError("Frozen Track B checkpoint integrity failure")
    device = torch.device(args.device)
    lease = resource_lease(
        args.resource_status,
        "gpu" if device.type == "cuda" else "cpu_validation",
        track="orn-eval-500-v1-transcriber",
        command=[sys.executable, *sys.argv],
        metadata={"rows": len(manifest["selected_rows"]), "locked_test": False},
    )
    lease.__enter__()
    atexit.register(lease.__exit__, None, None, None)
    try:
        model, frontend, decode, _payload = load_mel_checkpoint(checkpoint, device)
        decode = replace(decode, min_confidence=0.8)
        cache_root = output_dir / "transcriber-cache"
        prediction_rows = []
        cache_hits = 0
        for position, row in enumerate(manifest["selected_rows"], 1):
            key = _cache_key(
                row,
                checkpoint_sha,
                frontend.to_dict(),
                decode.to_dict(),
                window_frames=args.window_frames,
                overlap_frames=args.overlap_frames,
            )
            cache_path = cache_root / key[:2] / f"{key}.json"
            if cache_path.is_file():
                cached = _load_json(cache_path)
                if cached.get("cache_key") != key:
                    raise ValueError(f"Transcriber cache key mismatch: {cache_path}")
                notes = cached["notes"]
                cache_hits += 1
            else:
                audio_path = Path(
                    str(row["paths"]["performance_audio.wav"])
                )
                if sha256_file(audio_path) != row["hashes"]["performance_audio.wav"]:
                    raise ValueError(f"Audio hash mismatch: {row['sample']}")
                audio = load_audio_mono(audio_path, frontend.sample_rate)
                mel, normalization = extract_log_mel(
                    audio, frontend, device=device
                )
                probabilities = infer_mel_probabilities(
                    model,
                    mel,
                    device,
                    window_frames=args.window_frames,
                    overlap_frames=args.overlap_frames,
                    batch_size=args.batch_size,
                )
                notes = [
                    value.to_dict()
                    for value in decode_mel_notes(
                        probabilities,
                        midi_min=model.config.midi_min,
                        hop_sec=frontend.hop_sec,
                        config=decode,
                    )
                ]
                _atomic_json(
                    cache_path,
                    {
                        "schema_version": f"{SCHEMA_VERSION}-transcriber-cache",
                        "cache_key": key,
                        "sample": row["sample"],
                        "audio_sha256": row["hashes"]["performance_audio.wav"],
                        "checkpoint_sha256": checkpoint_sha,
                        "frontend": frontend.to_dict(),
                        "decode": decode.to_dict(),
                        "normalization": normalization,
                        "notes": notes,
                    },
                )
            prediction_rows.append(
                {
                    "sample": row["sample"],
                    "source_song_id": row["source_song_id"],
                    "provenance": row["provenance"],
                    "cache_key": key,
                    "notes": notes,
                }
            )
            if position == 1 or position % 25 == 0 or position == len(
                manifest["selected_rows"]
            ):
                print(
                    f"infer={position}/{len(manifest['selected_rows'])} "
                    f"cache_hits={cache_hits}",
                    flush=True,
                )
        predictions_path = output_dir / "predictions.jsonl"
        _atomic_jsonl(predictions_path, prediction_rows)
        freeze = {
            "schema_version": f"{SCHEMA_VERSION}-prediction-freeze",
            "created_utc": _utc(),
            "command": [sys.executable, *sys.argv],
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "predictions": str(predictions_path),
            "predictions_sha256": sha256_file(predictions_path),
            "rows": len(prediction_rows),
            "cache_hits": cache_hits,
            "cache_misses": len(prediction_rows) - cache_hits,
            "cache_key_fields": [
                "performance audio SHA-256",
                "checkpoint SHA-256",
                "frontend config",
                "decode config",
                "window/overlap frames",
            ],
            "frontend_config": frontend.to_dict(),
            "decode_config": decode.to_dict(),
            "inference_inputs": ["performance_audio.wav"],
            "score_input": False,
            "target_or_label_content_read": False,
            "pitch_output_space": "written",
            "locked_test_accessed": False,
            "production_weights_mutated": False,
        }
        _atomic_json(output_dir / "freeze_manifest.json", freeze)
    finally:
        atexit.unregister(lease.__exit__)
        lease.__exit__(None, None, None)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_targets(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return {
            str(row["sample"]): row
            for row in (json.loads(line) for line in stream if line.strip())
        }


def _candidates(notes: Sequence[Mapping[str, Any]]) -> tuple[JointCandidate, ...]:
    return tuple(
        JointCandidate(
            pitch=int(row["pitch"]),
            start=float(row["start"]),
            end=float(row["end"]),
            confidence=float(row.get("confidence", 1.0)),
        )
        for row in notes
    )


def _oracle_candidates(events: Sequence[JointEvent]) -> tuple[JointCandidate, ...]:
    return tuple(
        JointCandidate(
            pitch=event.pitch,
            start=event.start,
            end=event.end,
            confidence=1.0,
        )
        for event in events
    )


def _candidate_events(
    events: Sequence[JointCandidate],
) -> tuple[JointEvent, ...]:
    return tuple(
        JointEvent(
            pitch=event.pitch,
            start=event.start,
            end=event.end,
            score_span=None,
            relationship="extra",
            rendered_index=index,
            confidence=event.confidence,
        )
        for index, event in enumerate(events)
    )


def _transcription_prediction(event: JointEvent, index: int) -> JointEvent:
    if event.score_span is None:
        return replace(
            event,
            relationship="extra",
            copy_pass=0,
            rendered_index=index,
        )
    return replace(
        event,
        relationship="match",
        origin_relationship="match",
        copy_pass=0,
    )


def _transcription_target(event: JointEvent) -> JointEvent:
    if event.score_span is None:
        return event
    return replace(
        event,
        relationship="match",
        origin_relationship="match",
        copy_pass=0,
    )


def _fractional_prf(
    credit: float, predicted: int, gold: int
) -> dict[str, Any]:
    precision = credit / predicted if predicted else (1.0 if not gold else 0.0)
    recall = credit / gold if gold else (1.0 if not predicted else 0.0)
    return {
        "credit": credit,
        "predicted": predicted,
        "gold": gold,
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
    }


def _metric_counts(sample: JointMetricSample) -> tuple[float, int, int]:
    report = evaluate_joint_events(
        sample.predicted,
        sample.target,
        predicted_deletions=sample.predicted_deletions,
        target_deletions=sample.target_deletions,
        score_event_count=sample.score_event_count,
        tolerances_sec=(),
    )["official_note_wise"]
    return (
        float(report["credit"]),
        int(report["predicted"]),
        int(report["gold"]),
    )


def _bootstrap(
    counts: Sequence[tuple[float, int, int]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    if not counts:
        return {"replicates": 0, "lower_95": None, "median": None, "upper_95": None}
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        chosen = generator.integers(0, len(counts), len(counts))
        credit = sum(counts[index][0] for index in chosen)
        predicted = sum(counts[index][1] for index in chosen)
        gold = sum(counts[index][2] for index in chosen)
        values.append(_fractional_prf(credit, predicted, gold)["f1"])
    return {
        "replicates": replicates,
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _lcs(left: Sequence[int], right: Sequence[int]) -> int:
    row = [0] * (len(right) + 1)
    for item in left:
        previous = 0
        for column, other in enumerate(right, 1):
            saved = row[column]
            row[column] = (
                previous + 1
                if item == other
                else max(row[column], row[column - 1])
            )
            previous = saved
    return row[-1]


def _aggregate(samples: Sequence[JointMetricSample]) -> dict[str, Any]:
    return evaluate_joint_dataset(samples, tolerances_sec=())["aggregate"][
        "official_note_wise"
    ]


def _per_type(
    samples: Sequence[JointMetricSample],
) -> dict[str, dict[str, Any]]:
    output = {}
    for kind in ("match", "copy", "substitute", "extra"):
        selected = [
            JointMetricSample(
                predicted=tuple(
                    event
                    for event in sample.predicted
                    if ("copy" if event.is_copy else event.relationship) == kind
                ),
                target=tuple(
                    event
                    for event in sample.target
                    if ("copy" if event.is_copy else event.relationship) == kind
                ),
                source=sample.source,
                score_event_count=sample.score_event_count,
            )
            for sample in samples
        ]
        output[kind] = _aggregate(selected)
    return output


def _source_distribution(
    samples: Sequence[JointMetricSample],
    provenance_by_source: Mapping[str, str],
) -> dict[str, Any]:
    grouped: defaultdict[str, list[JointMetricSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.source].append(sample)
    metrics = {key: _aggregate(values) for key, values in grouped.items()}

    def summarize(keys: Sequence[str]) -> dict[str, Any]:
        values = [float(metrics[key]["f1"]) for key in keys]
        return {
            "sources": len(keys),
            "macro_f1": float(np.mean(values)) if values else None,
            "median_f1": float(np.median(values)) if values else None,
            "q05_f1": float(np.quantile(values, 0.05)) if values else None,
            "q25_f1": float(np.quantile(values, 0.25)) if values else None,
            "q75_f1": float(np.quantile(values, 0.75)) if values else None,
            "q95_f1": float(np.quantile(values, 0.95)) if values else None,
            "minimum_f1": min(values) if values else None,
            "maximum_f1": max(values) if values else None,
        }

    return {
        "all_source_songs": summarize(sorted(grouped)),
        "raw_score_source_songs": summarize(
            sorted(
                key
                for key in grouped
                if provenance_by_source.get(key) == "raw_score"
            )
        ),
        "random_generated_source_songs": summarize(
            sorted(
                key
                for key in grouped
                if provenance_by_source.get(key) == "random_generated"
            )
        ),
        "per_source": metrics,
    }


def run_evaluate(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    freeze_path = output_dir / "freeze_manifest.json"
    manifest = _load_json(manifest_path)
    freeze = _load_json(freeze_path)
    if freeze["manifest_sha256"] != sha256_file(manifest_path):
        raise ValueError("Prediction freeze belongs to another manifest")
    predictions_path = Path(str(freeze["predictions"]))
    if freeze["predictions_sha256"] != sha256_file(predictions_path):
        raise ValueError("Frozen prediction checksum mismatch")
    predictions = {
        str(row["sample"]): row for row in _read_jsonl(predictions_path)
    }
    selected_rows = {
        str(row["sample"]): row for row in manifest["selected_rows"]
    }
    if set(predictions) != set(selected_rows):
        raise ValueError("Prediction IDs do not equal frozen selected IDs")
    targets_path = Path(str(manifest["artifacts"]["canonical_targets"]["path"]))
    if (
        sha256_file(targets_path)
        != manifest["artifacts"]["canonical_targets"]["sha256"]
    ):
        raise ValueError("Canonical target cache checksum mismatch")
    targets = _read_targets(targets_path)
    if set(targets) != set(selected_rows):
        raise ValueError("Canonical target IDs do not equal selected IDs")

    combined_samples = []
    oracle_samples = []
    transcription_samples = []
    sequence_counts = [0.0, 0, 0]
    sequence_per_row = []
    duration_groups = {
        "lt_80ms": [0, 0],
        "lt_120ms": [0, 0],
        "lt_180ms": [0, 0],
        "ge_180ms": [0, 0],
    }
    predicted_same_pitch = target_same_pitch = 0
    predicted_total = target_total = 0
    per_row = []
    provenance_by_source = {}
    costs = GrammarCosts()
    for position, sample in enumerate(sorted(selected_rows), 1):
        row = selected_rows[sample]
        target_row = targets[sample]
        lineage = target_row["lineage"]
        if (
            hashlib.sha256(_canonical_bytes(lineage)).hexdigest()
            != row["target_lineage_sha256"]
        ):
            raise ValueError(f"Canonical target lineage mismatch: {sample}")
        score_path = Path(str(row["paths"]["verified_score.musicxml"]))
        if sha256_file(score_path) != row["hashes"]["verified_score.musicxml"]:
            raise ValueError(f"Verified score checksum mismatch: {sample}")
        index = ScoreEventIndex.from_musicxml(score_path, lineage)
        score = index.events
        target = index.rendered_events
        candidate = _candidates(predictions[sample]["notes"])
        oracle_candidate = _oracle_candidates(target)
        combined, _combined_grammar = decode_grammar_mapper(
            candidate, score, costs=costs
        )
        oracle, _oracle_grammar = decode_grammar_mapper(
            oracle_candidate, score, costs=costs
        )
        baseline, _baseline_deletions = current_note_aligner_baseline(
            candidate, score
        )
        transcription_predicted = tuple(
            _transcription_prediction(event, event_index)
            for event_index, event in enumerate(baseline)
        )
        transcription_target = tuple(
            _transcription_target(event) for event in target
        )
        source = str(row["source_song_id"])
        provenance_by_source[source] = str(row["provenance"])
        combined_sample = JointMetricSample(
            combined, target, source=source, score_event_count=len(score)
        )
        oracle_sample = JointMetricSample(
            oracle, target, source=source, score_event_count=len(score)
        )
        transcription_sample = JointMetricSample(
            transcription_predicted,
            transcription_target,
            source=source,
            score_event_count=len(score),
        )
        combined_samples.append(combined_sample)
        oracle_samples.append(oracle_sample)
        transcription_samples.append(transcription_sample)

        predicted_pitch = [event.pitch for event in candidate]
        target_pitch = [event.pitch for event in target]
        correct = _lcs(predicted_pitch, target_pitch)
        sequence_counts[0] += correct
        sequence_counts[1] += len(predicted_pitch)
        sequence_counts[2] += len(target_pitch)
        sequence_per_row.append((float(correct), len(candidate), len(target)))
        pairs = pair_exact_pitch_onset(
            _candidate_events(candidate), target, tolerance_sec=0.050
        )
        paired_target = {right for _left, right in pairs}
        for name, predicate in (
            ("lt_80ms", lambda value: value < 0.080),
            ("lt_120ms", lambda value: value < 0.120),
            ("lt_180ms", lambda value: value < 0.180),
            ("ge_180ms", lambda value: value >= 0.180),
        ):
            chosen = {
                event_index
                for event_index, event in enumerate(target)
                if predicate(event.end - event.start)
            }
            duration_groups[name][0] += len(chosen & paired_target)
            duration_groups[name][1] += len(chosen)
        predicted_same_pitch += sum(
            left.pitch == right.pitch
            for left, right in zip(candidate, candidate[1:])
        )
        target_same_pitch += sum(
            left.pitch == right.pitch
            for left, right in zip(target, target[1:])
        )
        predicted_total += len(candidate)
        target_total += len(target)
        per_row.append(
            {
                "sample": sample,
                "source_song_id": source,
                "provenance": row["provenance"],
                "predicted_notes": len(candidate),
                "target_notes": len(target),
                "score_events": len(score),
                "sequence_lcs": correct,
                "transcriber_canonical": _fractional_prf(
                    *_metric_counts(transcription_sample)
                ),
                "oracle_mapper": _fractional_prf(*_metric_counts(oracle_sample)),
                "combined": _fractional_prf(*_metric_counts(combined_sample)),
            }
        )
        if position == 1 or position % 25 == 0 or position == len(selected_rows):
            print(f"evaluate={position}/{len(selected_rows)}", flush=True)

    main_samples = {
        "transcriber_canonical_note_wise": transcription_samples,
        "oracle_note_mapper_canonical_note_wise": oracle_samples,
        "combined_canonical_note_wise": combined_samples,
    }
    main_metrics = {}
    for offset, (name, samples) in enumerate(main_samples.items()):
        counts = [_metric_counts(sample) for sample in samples]
        main_metrics[name] = {
            **_aggregate(samples),
            "bootstrap_95": _bootstrap(
                counts,
                seed=args.seed + offset,
                replicates=args.bootstrap_replicates,
            ),
            "by_type": _per_type(samples),
            "source_song_distribution": _source_distribution(
                samples, provenance_by_source
            ),
        }
    ornament_totals = {
        key: sum(
            int((row.get("ornament_realization") or {}).get(key) or 0)
            for row in selected_rows.values()
        )
        for key in (
            "rendered_events",
            "performed_events",
            "unmapped_rendered_events",
            "outside_track_b_written_pitch_range",
        )
    }
    ornament_totals["bundles_with_rendered_polyphony"] = sum(
        int(
            (row.get("ornament_realization") or {}).get(
                "max_simultaneous_rendered_events", 0
            )
            > 1
        )
        for row in selected_rows.values()
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "phase": "external_evaluation",
        "created_utc": _utc(),
        "command": [sys.executable, *sys.argv],
        "data": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "selected_rows": len(selected_rows),
            "selected_source_songs": len(
                {row["source_song_id"] for row in selected_rows.values()}
            ),
            "provenance": dict(
                sorted(
                    Counter(
                        str(row["provenance"])
                        for row in selected_rows.values()
                    ).items()
                )
            ),
        },
        "metric": {
            "schema_version": "align-note-wise-score-event-metric-v1",
            "unit": "canonical score-note/event identity",
            "matching": (
                "exclusive one-to-one; exact location+type=1.0, exact "
                "location+wrong type=0.5, wrong location=0"
            ),
            "ties": "collapsed by ScoreEventIndex",
            "extras": "exclusive rendered_index identity",
            "timestamp_metrics": "diagnostic_only",
        },
        "models": manifest["models"],
        "score_agnostic_transcriber_diagnostic": {
            "pitch_sequence_lcs": {
                **_fractional_prf(*sequence_counts),
                "bootstrap_95": _bootstrap(
                    sequence_per_row,
                    seed=args.seed + 10,
                    replicates=args.bootstrap_replicates,
                ),
            },
            "count_ratio": predicted_total / max(target_total, 1),
            "short_note_recall_at_50ms_onset": {
                name: {
                    "matched": values[0],
                    "support": values[1],
                    "recall": values[0] / max(values[1], 1),
                }
                for name, values in duration_groups.items()
            },
            "same_pitch_split_rearticulation": {
                "predicted_adjacent_same_pitch": predicted_same_pitch,
                "predicted_rate": predicted_same_pitch / max(predicted_total, 1),
                "target_adjacent_same_pitch": target_same_pitch,
                "target_rate": target_same_pitch / max(target_total, 1),
                "note": (
                    "Descriptive only: ornament-expanded targets can overlap, "
                    "so adjacent order is not a universal rearticulation identity."
                ),
            },
            "timestamp_status": "diagnostic_only",
        },
        **main_metrics,
        "ornament_target_audit": {
            **ornament_totals,
            "canonical_scoreability": "available for every selected row",
            "caveats": [
                (
                    "Renderer-realized ornament notes without performed-note "
                    "lineage are canonical rendered EXTRA events. They are valid "
                    "exclusive identities, but they are not necessarily injected "
                    "performance errors."
                ),
                (
                    "Overlapping ornament realizations make some targets "
                    "polyphonic while frozen Track B emits one pitch per frame; "
                    "those rows are scoreable but architecturally out of domain."
                ),
            ],
        },
        "per_row": per_row,
        "isolation": {
            "external_evaluation_only": True,
            "ORN_used_for_tuning": False,
            "threshold_or_decoder_changed": False,
            "selected_min_confidence": 0.8,
            "locked_test_accessed": False,
            "production_weights_mutated": False,
        },
    }
    report_path = output_dir / "report.json"
    _atomic_json(report_path, report)

    checked_files = {}
    source_unchanged = True
    for sample, row in selected_rows.items():
        actual = {
            name: sha256_file(Path(str(path)))
            for name, path in row["paths"].items()
            if name in row["hashes"]
        }
        unchanged = actual == row["hashes"]
        source_unchanged &= unchanged
        checked_files[sample] = {"unchanged": unchanged, "hashes": actual}
    integrity_path = output_dir / "integrity.json"
    integrity = {
        "schema_version": f"{SCHEMA_VERSION}-integrity",
        "created_utc": _utc(),
        "manifest_sha256": sha256_file(manifest_path),
        "freeze_manifest_sha256": sha256_file(freeze_path),
        "predictions_sha256": sha256_file(predictions_path),
        "canonical_targets_sha256": sha256_file(targets_path),
        "report_sha256": sha256_file(report_path),
        "checkpoint_sha256": sha256_file(
            Path(str(manifest["models"]["transcriber_checkpoint"]["path"]))
        ),
        "mapper_reference_report_sha256": sha256_file(
            Path(str(manifest["models"]["mapper"]["reference_report"]))
        ),
        "mapper_status_sha256": sha256_file(
            Path(str(manifest["models"]["mapper"]["status"]))
        ),
        "selected_source_files_unchanged": source_unchanged,
        "selected_source_file_checks": checked_files,
        "passed": source_unchanged,
        "ORN_used_for_tuning": False,
        "locked_test_accessed": False,
        "production_weights_mutated": False,
    }
    _atomic_json(integrity_path, integrity)
    status = {
        "schema_version": f"{SCHEMA_VERSION}-status",
        "status": "complete" if integrity["passed"] else "integrity_failed",
        "completed_utc": _utc(),
        "rows": len(selected_rows),
        "source_songs": report["data"]["selected_source_songs"],
        "provenance": report["data"]["provenance"],
        "scores": {
            "score_agnostic_sequence_f1_diagnostic": report[
                "score_agnostic_transcriber_diagnostic"
            ]["pitch_sequence_lcs"]["f1"],
            "transcriber_canonical_f1": report[
                "transcriber_canonical_note_wise"
            ]["f1"],
            "oracle_mapper_canonical_f1": report[
                "oracle_note_mapper_canonical_note_wise"
            ]["f1"],
            "combined_canonical_f1": report["combined_canonical_note_wise"]["f1"],
        },
        "commands": {
            "audit": manifest["command"],
            "infer": freeze["command"],
            "evaluate": report["command"],
        },
        "artifacts": {
            "manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "freeze_manifest": {
                "path": str(freeze_path),
                "sha256": sha256_file(freeze_path),
            },
            "predictions": {
                "path": str(predictions_path),
                "sha256": sha256_file(predictions_path),
            },
            "report": {"path": str(report_path), "sha256": sha256_file(report_path)},
            "integrity": {
                "path": str(integrity_path),
                "sha256": sha256_file(integrity_path),
            },
        },
        "no_tuning": True,
        "locked_test_accessed": False,
        "production_weights_mutated": False,
    }
    _atomic_json(output_dir / "STATUS.json", status)
    print(json.dumps(status["scores"], indent=2), flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("audit", "refreeze", "infer", "evaluate")
    )
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--source-root", type=Path, default=Path("E:/outputRaw_orn_10k")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "runs/joint-outputraw-full-v1/orn-eval-500-v1"
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--mapper-report", type=Path, default=DEFAULT_MAPPER_REPORT)
    parser.add_argument("--mapper-status", type=Path, default=DEFAULT_MAPPER_STATUS)
    parser.add_argument(
        "--resource-status", type=Path, default=DEFAULT_RESOURCE_STATUS
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--window-frames", type=int, default=2048)
    parser.add_argument("--overlap-frames", type=int, default=512)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")
    args.repo = args.repo.resolve()
    args.output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else args.repo / args.output_dir
    )
    args.checkpoint = (
        args.checkpoint
        if args.checkpoint.is_absolute()
        else args.repo / args.checkpoint
    )
    args.mapper_report = (
        args.mapper_report
        if args.mapper_report.is_absolute()
        else args.repo / args.mapper_report
    )
    args.mapper_status = (
        args.mapper_status
        if args.mapper_status.is_absolute()
        else args.repo / args.mapper_status
    )
    args.resource_status = (
        args.resource_status
        if args.resource_status.is_absolute()
        else args.repo / args.resource_status
    )
    if args.phase == "audit":
        run_audit(args)
    elif args.phase == "refreeze":
        run_refreeze(args)
    elif args.phase == "infer":
        run_infer(args)
    else:
        with resource_lease(
            args.resource_status,
            "cpu_validation",
            track="orn-eval-500-v1-scoring",
            command=[sys.executable, *sys.argv],
            metadata={"locked_test": False},
        ):
            run_evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
