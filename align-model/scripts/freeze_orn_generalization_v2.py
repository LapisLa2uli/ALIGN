"""Freeze the globally reconstructed, prior-release-disjoint ORN v2 release."""

from __future__ import annotations

import argparse
import atexit
import gzip
import hashlib
import json
import os
import secrets
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import audit_training_data as audit
import freeze_orn_generalization_v1 as v1
from alignmodel.joint.global_ornament_lineage_v2 import (
    SCHEMA_VERSION as REPAIR_VERSION,
    reconstruct_global_ornament_lineage,
)
from alignmodel.joint.identity_crf_v1 import (
    IdentityCandidate,
    build_identity_lattice,
    exact_identity_round_trip,
)
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.packed_data import sha256_file
from alignmodel.training_resources import resource_lease


SCHEMA_VERSION = "align-orn-generalization-release-v2"
POLICY_VERSION = "align-orn-generalization-v2-policy"
DEFAULT_OUTPUT = Path(
    "runs/joint-outputraw-full-v1/orn-generalization-v2"
)
DEFAULT_UPSTREAM = Path(
    "runs/joint-outputraw-full-v1/orn-eval-500-v1/manifest.json"
)
DEFAULT_V1 = Path(
    "runs/joint-outputraw-full-v1/orn-generalization-v1/release_manifest.json"
)
DEFAULT_OUTPUTRAW_TARGETS = Path(
    "data-audit/joint-outputraw-full-v1/canonical_dev_targets.sqlite"
)
DEFAULT_OUTPUTRAW_SEAL = Path(
    "data-audit/joint-outputraw-full-v1/LOCKBOX_SEALED.json"
)
DEFAULT_RESOURCE_STATUS = Path("runs/TRAINING_RESOURCE_STATUS.json")
DEFAULT_SEED = 20260919
SPLIT_FRACTIONS = {
    "lockbox": (0.00, 0.05),
    "open_validation": (0.05, 0.15),
    "calibration": (0.15, 0.20),
    "train": (0.20, 1.00),
}
SPLITS = ("train", "calibration", "open_validation", "lockbox")
MIN_RAW_LOCKBOX_GROUPS = 20
REQUIRED_FILES = v1.REQUIRED_FILES
AAD = f"{SCHEMA_VERSION}:lockbox-targets".encode("ascii")


def _policy_document(args: argparse.Namespace) -> dict[str, Any]:
    repair_path = (
        args.repo
        / "src"
        / "alignmodel"
        / "joint"
        / "global_ornament_lineage_v2.py"
    )
    identity_path = (
        args.repo
        / "src"
        / "alignmodel"
        / "joint"
        / "identity_crf_v1.py"
    )
    return {
        "schema_version": POLICY_VERSION,
        "created_utc": v1._utc(),
        "seed": args.seed,
        "selection_outcome_blind": True,
        "source_root": "E:\\outputRaw_orn_10k",
        "prior_exclusions": {
            "orn500": {
                "manifest": str(args.upstream_manifest.resolve()),
                "manifest_sha256": sha256_file(args.upstream_manifest),
                "scope": (
                    "all selected rows plus source-song, verified-score/content, "
                    "and performance-audio groups"
                ),
                "prediction_or_outcome_files_read": False,
            },
            "orn_generalization_v1": {
                "manifest": str(args.v1_manifest.resolve()),
                "manifest_sha256": sha256_file(args.v1_manifest),
                "scope": (
                    "every development row/group and public sealed-lockbox "
                    "verified-score/audio group; v1 targets/key are forbidden"
                ),
            },
            "outputraw": {
                "development_targets": str(args.outputraw_targets.resolve()),
                "splits": ["train", "val"],
                "original_test_read": False,
            },
        },
        "repair": {
            "version": REPAIR_VERSION,
            "module": str(repair_path),
            "module_sha256": sha256_file(repair_path),
            "rules": [
                "parse generator performance MusicXML and non-grace performed lineage",
                "collapse true notated tie chains before assignment",
                "expand grace/trill/mordent/turn control flow exactly as the renderer",
                "globally align raw MIDI and renderer template one-to-one by exact written pitch",
                "require zero raw/template edits and every performed identity mapped",
                "preserve planted EXTRA/copy lineage separately from renderer ornament EXTRA",
                "project to tie-aware verified-score canonical spans",
                "require zero backward identity steps within each replay pass",
                "preserve raw MIDI temporal order and explicit rendered identity",
                "fail closed on every ambiguity or failed exact round trip",
            ],
            "acoustic_predictions_used": False,
        },
        "eligibility": {
            "required_files": list(REQUIRED_FILES),
            "schemas": {
                "metadata": "1.2",
                "labels": "1.2",
                "source_note_map": "1.0",
                "corrected_lineage": "2.0",
            },
            "pitch_policy": {
                "audio_pitch_space": "sounding",
                "midi_pitch_space": "sounding",
                "effective_audio_transpose": 2,
                "sounding_transpose": -2,
            },
            "valid_label_spans_required": True,
            "repeat_metadata_lineage_consistency_required": True,
            "identity_crf_gold_coverage_required": True,
            "exact_identity_round_trip_required": True,
        },
        "grouping": {
            "raw": "authoritative source-score SHA-256",
            "generated": (
                "connected component of verified-score, clean content, "
                "transposition-invariant content, performed lineage, and audio SHA-256"
            ),
            "duplicates_never_cross_splits": True,
        },
        "split_fractions_by_group": SPLIT_FRACTIONS,
        "raw_lockbox_minimum_independent_sources": MIN_RAW_LOCKBOX_GROUPS,
        "metric": {
            "unit": "canonical score-note/event identity",
            "matching": "exclusive one-to-one",
            "exact_identity_and_type": 1.0,
            "exact_identity_wrong_type": 0.5,
            "wrong_identity": 0.0,
            "timestamp_metrics": "diagnostic_only",
        },
        "gate": {
            "combined_open_validation_f1": 0.85,
            "lockbox_openings": 1,
            "post_test_tuning": False,
        },
        "identity_crf_module": {
            "path": str(identity_path),
            "sha256": sha256_file(identity_path),
        },
        "locked_before_membership_selection": True,
        "production_weights_mutated": False,
        "paper_updated": False,
    }


def freeze_policy(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "POLICY.json"
    freeze = args.output_dir / "POLICY_FREEZE.json"
    if path.exists() or freeze.exists():
        raise FileExistsError("ORN v2 policy is already frozen")
    document = _policy_document(args)
    v1._atomic_json(path, document)
    v1._atomic_json(
        freeze,
        {
            "schema_version": f"{POLICY_VERSION}-freeze",
            "frozen_utc": v1._utc(),
            "policy": v1._artifact(path),
            "membership_selected": False,
            "predictions_or_outcomes_read": False,
        },
    )
    print(json.dumps(v1._artifact(path), indent=2), flush=True)


def _verify_policy(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    path = args.output_dir / "POLICY.json"
    freeze_path = args.output_dir / "POLICY_FREEZE.json"
    document = v1._load_json(path)
    freeze = v1._load_json(freeze_path)
    digest = sha256_file(path)
    if freeze["policy"]["sha256"] != digest:
        raise ValueError("Frozen ORN v2 policy hash mismatch")
    if int(document["seed"]) != int(args.seed):
        raise ValueError("Runtime seed differs from frozen policy")
    if (
        document["prior_exclusions"]["orn500"]["manifest_sha256"]
        != sha256_file(args.upstream_manifest)
        or document["prior_exclusions"]["orn_generalization_v1"][
            "manifest_sha256"
        ]
        != sha256_file(args.v1_manifest)
    ):
        raise ValueError("Prior exclusion artifact differs from frozen policy")
    repair_path = (
        args.repo
        / "src"
        / "alignmodel"
        / "joint"
        / "global_ornament_lineage_v2.py"
    )
    identity_path = (
        args.repo
        / "src"
        / "alignmodel"
        / "joint"
        / "identity_crf_v1.py"
    )
    if (
        document["repair"]["module_sha256"] != sha256_file(repair_path)
        or document["identity_crf_module"]["sha256"]
        != sha256_file(identity_path)
    ):
        raise ValueError("Frozen repair/lattice implementation hash mismatch")
    return document, digest


def _prior_identities(
    args: argparse.Namespace,
) -> dict[str, set[str]]:
    output: defaultdict[str, set[str]] = defaultdict(set)
    orn500 = v1._load_json(args.upstream_manifest)
    for row in orn500["selected_rows"]:
        output["sample"].add(str(row["sample"]))
        output["source_song_id"].add(str(row["source_song_id"]))
        if row.get("source_score_sha256"):
            output["source_score_sha256"].add(
                str(row["source_score_sha256"])
            )
        if row.get("content_fingerprint"):
            output["content_fingerprint"].add(
                str(row["content_fingerprint"])
            )
        for name, value in (row.get("hashes") or {}).items():
            output[f"hash:{name}"].add(str(value))
    release_v1 = v1._load_json(args.v1_manifest)
    for split in ("train", "calibration", "open_validation"):
        for row in release_v1["splits"]["development"][split]:
            output["sample"].add(str(row["sample"]))
            output["source_song_id"].add(str(row["source_song_id"]))
            if row.get("source_score_sha256"):
                output["source_score_sha256"].add(
                    str(row["source_score_sha256"])
                )
            for name in (
                "clean_fingerprint",
                "near_lineage_hash",
                "lineage_hash",
                "audio_lineage_hash",
            ):
                output[name].add(str(row[name]))
            for name, value in row["source_hashes"].items():
                output[f"hash:{name}"].add(str(value))
    lockbox_manifest_path = Path(
        release_v1["splits"]["lockbox_manifest"]["path"]
    )
    if (
        sha256_file(lockbox_manifest_path)
        != release_v1["splits"]["lockbox_manifest"]["sha256"]
    ):
        raise ValueError("v1 public lockbox manifest mismatch")
    lockbox = v1._load_json(lockbox_manifest_path)
    for row in lockbox["inputs"]:
        output["hash:performance_audio.wav"].add(row["audio"]["sha256"])
        output["hash:verified_score.musicxml"].add(row["score"]["sha256"])
        output["leakage_group"].add(row["leakage_group"])
    outputraw = v1._outputraw_development_identities(
        args.outputraw_targets, args.outputraw_seal
    )
    for name, values in outputraw.items():
        if name != "split":
            output[f"outputraw:{name}"].update(values)
    return dict(output)


def _fast_prior_reasons(
    upstream_row: Mapping[str, Any],
    prior: Mapping[str, set[str]],
) -> list[str]:
    reasons = []
    if str(upstream_row["sample"]) in prior.get("sample", set()):
        reasons.append("prior_sample")
    if str(upstream_row.get("source_song_id") or "") in prior.get(
        "source_song_id", set()
    ):
        reasons.append("prior_source_song")
    if (
        upstream_row.get("source_score_sha256")
        and str(upstream_row["source_score_sha256"])
        in prior.get("source_score_sha256", set())
    ):
        reasons.append("prior_raw_source_score")
    if (
        upstream_row.get("content_fingerprint")
        and str(upstream_row["content_fingerprint"])
        in prior.get("content_fingerprint", set())
    ):
        reasons.append("prior_content")
    hashes = upstream_row.get("hashes") or {}
    for name in ("verified_score.musicxml", "performance_audio.wav"):
        if hashes.get(name) in prior.get(f"hash:{name}", set()):
            reasons.append(f"prior_{name}")
        if hashes.get(name) in prior.get(f"outputraw:hash:{name}", set()):
            reasons.append(f"outputraw_{name}")
    return sorted(set(reasons))


def _audit_row(
    sample_dir: Path,
    source_root: Path,
    hash_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    issues = v1.RecordingIssueLog()
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
    reasons = []

    def reject(code: str, detail: str) -> None:
        reasons.append({"code": code, "detail": detail})

    missing = [name for name in REQUIRED_FILES if not (sample_dir / name).is_file()]
    if missing:
        reject("required_file_missing", ", ".join(missing))
    try:
        metadata = v1._load_json(sample_dir / "metadata.json")
        labels = v1._load_json(sample_dir / "labels.json")
        original = v1._load_json(sample_dir / "note_map.json")
    except Exception as exc:
        return {
            "sample": sample_dir.name,
            "sample_dir": str(sample_dir),
            "eligible": False,
            "exclusion_reasons": [
                {"code": "required_json_invalid", "detail": str(exc)}
            ],
        }
    if metadata.get("schema_version") != "1.2":
        reject("metadata_schema_invalid", repr(metadata.get("schema_version")))
    if labels.get("schema_version") != "1.2":
        reject("labels_schema_invalid", repr(labels.get("schema_version")))
    if original.get("schema_version") != "1.0":
        reject("source_note_map_schema_invalid", repr(original.get("schema_version")))
    if not result.get("note_map_valid"):
        reject("source_note_map_invalid", "structural source audit failed")
    if int(result.get("invalid_label_count") or 0):
        reject("invalid_label_spans", str(result["invalid_label_count"]))
    if not result.get("repeat_consistent", True):
        reject("repeat_representation_inconsistent", "metadata/COPY disagreement")
    expected_pitch = {
        "audio_pitch_space": "sounding",
        "midi_pitch_space": "sounding",
        "effective_audio_transpose": 2,
        "sounding_transpose": -2,
    }
    actual_pitch = {
        name: result.get(name) for name in expected_pitch
    }
    if actual_pitch != expected_pitch:
        reject(
            "bb_written_pitch_policy_invalid",
            f"expected={expected_pitch} actual={actual_pitch}",
        )
    wav = audit._read_wave_info(sample_dir / "performance_audio.wav")
    midi = audit._midi_events(sample_dir / "performance_audio.mid")
    if wav is None or float(wav.get("duration_sec") or 0) <= 0:
        reject("performance_audio_invalid", "unreadable or empty")
    if midi is None or not midi.get("notes"):
        reject("performance_midi_invalid", "unreadable or empty")
    if wav is not None and midi is not None:
        last = max(float(row["end"]) for row in midi["notes"])
        if last > float(wav["duration_sec"]) + 0.15:
            reject(
                "midi_event_outside_audio",
                f"event={last:.6f} wav={wav['duration_sec']:.6f}",
            )
    hashes = {}
    for name in REQUIRED_FILES:
        path = sample_dir / name
        if path.is_file():
            hashes[name] = audit._sha256(path, hash_cache)
    corrected = None
    stats = {}
    if midi is not None:
        try:
            shift = audit._inferred_midi_shift(original, midi, metadata)
            reconstruction = reconstruct_global_ornament_lineage(
                original,
                sample_dir / "performance_score.musicxml",
                sample_dir / "verified_score.musicxml",
                midi,
                written_shift=shift,
            )
            corrected = reconstruction.lineage
            stats = reconstruction.stats
            index = ScoreEventIndex.from_musicxml(
                sample_dir / "verified_score.musicxml", corrected
            )
            lattice = build_identity_lattice(
                tuple(
                    IdentityCandidate(
                        event.pitch, event.start, event.end, 1.0
                    )
                    for event in index.rendered_events
                ),
                index.events,
                sample_dir / "verified_score.musicxml",
                targets=index.rendered_events,
                target_deletions=index.deleted_event_indices,
                max_negative_hypotheses=0,
            )
            round_trip = exact_identity_round_trip(lattice)
            if not round_trip["passed"]:
                reject("identity_crf_round_trip_failed", repr(round_trip))
            stats["identity_crf_gold_hypotheses"] = sum(
                value.is_gold_compatible for value in lattice.hypotheses
            )
            stats["identity_crf_round_trip"] = round_trip
        except Exception as exc:
            reject(
                "global_ornament_reconstruction_failed",
                f"{type(exc).__name__}: {exc}",
            )
    verified_sha = hashes.get("verified_score.musicxml", "")
    try:
        provenance, source_song_id, source_score_sha = v1._source_identity(
            metadata, verified_sha, audit, hash_cache
        )
    except Exception as exc:
        provenance, source_song_id, source_score_sha = (
            "invalid",
            f"invalid:{sample_dir.name}",
            None,
        )
        reject("source_identity_invalid", str(exc))
    fingerprints = (
        v1._lineage_fingerprints(
            corrected, hashes.get("performance_audio.wav", "")
        )
        if corrected is not None
        else {}
    )
    return {
        "sample": sample_dir.name,
        "sample_dir": str(sample_dir.resolve()),
        "eligible": not reasons,
        "exclusion_reasons": reasons,
        "provenance": provenance,
        "source_song_id": source_song_id,
        "source_score_sha256": source_score_sha,
        "duration_sec": result.get("duration_sec"),
        "source_hashes": hashes,
        **fingerprints,
        "target_lineage_sha256": (
            hashlib.sha256(v1._canonical_bytes(corrected)).hexdigest()
            if corrected is not None
            else None
        ),
        "target_stats": stats,
        "_lineage": corrected,
    }


def _fraction(seed: int, group_id: str) -> float:
    return int(v1._stable(seed, group_id)[:16], 16) / float(16**16)


def _assign(
    groups: Mapping[str, Sequence[dict[str, Any]]],
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    raw_groups = [
        group_id
        for group_id, rows in groups.items()
        if any(row["provenance"] == "raw" for row in rows)
    ]
    raw_lockbox = len(raw_groups) >= MIN_RAW_LOCKBOX_GROUPS
    assigned = {split: [] for split in SPLITS}
    for group_id, rows in sorted(groups.items()):
        value = _fraction(seed, group_id)
        split = next(
            name
            for name in ("lockbox", "open_validation", "calibration", "train")
            if SPLIT_FRACTIONS[name][0] <= value < SPLIT_FRACTIONS[name][1]
        )
        if (
            any(row["provenance"] == "raw" for row in rows)
            and not raw_lockbox
            and split != "train"
        ):
            split = "train"
        for row in rows:
            row["split"] = split
            row["leakage_group"] = group_id
            assigned[split].append(row)
    for split in assigned:
        assigned[split].sort(
            key=lambda row: (
                v1._stable(seed + len(split), row["sample"]),
                row["sample"],
            )
        )
    if any(not assigned[split] for split in SPLITS):
        raise ValueError(
            f"Frozen v2 split is empty: "
            f"{ {name: len(rows) for name, rows in assigned.items()} }"
        )
    return assigned, {
        "raw_independent_groups": len(raw_groups),
        "raw_lockbox_allowed": raw_lockbox,
        "raw_lockbox_limitation": (
            None
            if raw_lockbox
            else (
                "Fewer than 20 prior-disjoint raw source scores remained; "
                "all raw rows were forced to train and the lockbox is generated-only."
            )
        ),
    }


def _encrypt(rows: Sequence[Mapping[str, Any]], key: bytes) -> bytes:
    plaintext = gzip.compress(
        b"".join(v1._canonical_bytes(row) + b"\n" for row in rows),
        compresslevel=9,
        mtime=0,
    )
    nonce = secrets.token_bytes(12)
    return b"ORNLBX2\0" + nonce + AESGCM(key).encrypt(nonce, plaintext, AAD)


def _decrypt(value: bytes, key: bytes) -> list[dict[str, Any]]:
    if not value.startswith(b"ORNLBX2\0"):
        raise ValueError("Invalid v2 lockbox envelope")
    plaintext = AESGCM(key).decrypt(value[8:20], value[20:], AAD)
    return [
        json.loads(line)
        for line in gzip.decompress(plaintext).splitlines()
        if line.strip()
    ]


def build(args: argparse.Namespace) -> None:
    if (args.output_dir / "FREEZE.json").exists():
        raise FileExistsError("ORN v2 release is already frozen")
    policy, policy_sha = _verify_policy(args)
    upstream = v1._load_json(args.upstream_manifest)
    inventory = v1._read_upstream_inventory(upstream)
    source_root = Path(upstream["source_root"])
    prior = _prior_identities(args)
    partial_path = args.output_dir / "audit_partial.jsonl"
    completed = v1._load_partial(partial_path)
    cache_path = args.output_dir / "audit_hash_cache.json"
    bootstrap_cache = Path(upstream["artifacts"]["audit_hash_cache"]["path"])
    if cache_path.is_file():
        hash_cache = v1._load_json(cache_path)
    elif (
        bootstrap_cache.is_file()
        and sha256_file(bootstrap_cache)
        == upstream["artifacts"]["audit_hash_cache"]["sha256"]
    ):
        hash_cache = v1._load_json(bootstrap_cache)
    else:
        hash_cache = {}
    excluded_fast = {}
    candidates = []
    for row in inventory:
        reasons = _fast_prior_reasons(row, prior)
        if reasons:
            excluded_fast[row["sample"]] = reasons
        else:
            candidates.append(row)
    lease = resource_lease(
        args.resource_status,
        "cpu_validation",
        track="orn-generalization-v2-audit-freeze",
        command=[sys.executable, *sys.argv],
        metadata={
            "candidate_rows": len(candidates),
            "locked_test": False,
            "v1_lockbox_targets_read": False,
        },
    )
    lease.__enter__()
    atexit.register(lease.__exit__, None, None, None)
    try:
        for position, upstream_row in enumerate(candidates, 1):
            sample = upstream_row["sample"]
            if sample in completed:
                continue
            row = _audit_row(
                Path(upstream_row["sample_dir"]), source_root, hash_cache
            )
            outputraw_matches = v1._outputraw_matches(
                row,
                {
                    name.removeprefix("outputraw:"): values
                    for name, values in prior.items()
                    if name.startswith("outputraw:")
                },
            )
            row["outputraw_matches"] = outputraw_matches
            if outputraw_matches:
                row["eligible"] = False
                row["exclusion_reasons"].append(
                    {
                        "code": "outputraw_development_overlap",
                        "detail": ", ".join(outputraw_matches),
                    }
                )
            completed[sample] = row
            v1._append_partial(partial_path, row)
            if position % args.checkpoint_every == 0:
                v1._atomic_json(cache_path, hash_cache)
                print(
                    f"audit={position}/{len(candidates)} "
                    f"eligible={sum(value.get('eligible', False) for value in completed.values())}",
                    flush=True,
                )
        v1._atomic_json(cache_path, hash_cache)
        eligible = [row for row in completed.values() if row.get("eligible")]
        groups = v1._group_records(eligible)
        assigned, assignment = _assign(groups, args.seed)
        leakage = v1._split_overlap(assigned)
        if not leakage["passed"]:
            raise RuntimeError("ORN v2 split leakage detected")
        development_targets = []
        development_manifest = {
            split: [] for split in SPLITS if split != "lockbox"
        }
        ordinal = 0
        for split in ("train", "calibration", "open_validation"):
            for row in assigned[split]:
                development_targets.append(
                    {
                        "ordinal": ordinal,
                        "split": split,
                        "sample": row["sample"],
                        "sample_dir": row["sample_dir"],
                        "source_hashes": row["source_hashes"],
                        "target_lineage_sha256": row[
                            "target_lineage_sha256"
                        ],
                        "representation_version": REPAIR_VERSION,
                        "lineage": row["_lineage"],
                    }
                )
                public = {
                    key: value
                    for key, value in row.items()
                    if not key.startswith("_") and key != "exclusion_reasons"
                }
                public["target_record"] = ordinal
                development_manifest[split].append(public)
                ordinal += 1
        target_path = args.output_dir / "development_targets.jsonl.gz"
        v1._atomic_jsonl_gz(target_path, development_targets)
        key = v1._create_or_load_key(args.lockbox_key)
        lockbox_dir = args.output_dir / "lockbox"
        input_dir = lockbox_dir / "inputs"
        lockbox_public = []
        lockbox_private = []
        for position, row in enumerate(assigned["lockbox"]):
            row_id = hashlib.sha256(
                f"{SCHEMA_VERSION}:{args.seed}:{row['leakage_group']}:{row['sample']}".encode()
            ).hexdigest()[:24]
            audio = input_dir / f"{row_id}.wav"
            score = input_dir / f"{row_id}.musicxml"
            if not audio.is_file():
                v1._atomic_copy(
                    Path(row["sample_dir"]) / "performance_audio.wav", audio
                )
            if not score.is_file():
                v1._atomic_copy(
                    Path(row["sample_dir"]) / "verified_score.musicxml", score
                )
            lockbox_public.append(
                {
                    "ordinal": position,
                    "row_id": row_id,
                    "provenance": row["provenance"],
                    "leakage_group": row["leakage_group"],
                    "duration_sec": row["duration_sec"],
                    "audio": v1._artifact(audio),
                    "score": v1._artifact(score),
                }
            )
            lockbox_private.append(
                {
                    "ordinal": position,
                    "row_id": row_id,
                    "sample": row["sample"],
                    "sample_dir": row["sample_dir"],
                    "source_hashes": row["source_hashes"],
                    "target_lineage_sha256": row[
                        "target_lineage_sha256"
                    ],
                    "representation_version": REPAIR_VERSION,
                    "lineage": row["_lineage"],
                }
            )
        encrypted = _encrypt(lockbox_private, key)
        if _decrypt(encrypted, key) != lockbox_private:
            raise RuntimeError("ORN v2 lockbox encryption round trip failed")
        encrypted_path = lockbox_dir / "targets.aes256gcm"
        v1._atomic_bytes(encrypted_path, encrypted)
        lockbox_manifest_path = lockbox_dir / "manifest.json"
        v1._atomic_json(
            lockbox_manifest_path,
            {
                "schema_version": f"{SCHEMA_VERSION}-lockbox-manifest",
                "created_utc": v1._utc(),
                "rows": len(lockbox_public),
                "inputs": lockbox_public,
                "target_envelope": {
                    **v1._artifact(encrypted_path),
                    "algorithm": "AES-256-GCM",
                    "aad": AAD.decode(),
                    "key_sha256": hashlib.sha256(key).hexdigest(),
                    "key_stored_outside_repository": True,
                },
                "target_content_public": False,
                "source_bundle_paths_public": False,
            },
        )
        exclusions = [
            {
                "sample": row["sample"],
                "stage": "prior_release_or_exposure_exclusion",
                "reasons": excluded_fast[row["sample"]],
            }
            for row in inventory
            if row["sample"] in excluded_fast
        ]
        exclusions.extend(
            {
                "sample": row["sample"],
                "stage": "v2_global_audit",
                "reasons": row["exclusion_reasons"],
            }
            for row in completed.values()
            if not row.get("eligible")
        )
        exclusions_path = args.output_dir / "exclusions.jsonl.gz"
        v1._atomic_jsonl_gz(exclusions_path, exclusions)
        leakage_path = args.output_dir / "leakage_report.json"
        v1._atomic_json(
            leakage_path,
            {
                "schema_version": f"{SCHEMA_VERSION}-leakage",
                "new_split_overlap": leakage,
                "prior_excluded_rows": len(excluded_fast),
                "outputraw_overlap_in_admitted": 0,
                "v1_lockbox_target_or_key_read": False,
                "passed": leakage["passed"],
            },
        )
        split_counts = {
            split: len(rows) for split, rows in assigned.items()
        }
        source_counts = {
            split: len({row["leakage_group"] for row in rows})
            for split, rows in assigned.items()
        }
        input_files = [path for path in input_dir.iterdir() if path.is_file()]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "release": "orn-generalization-v2",
            "phase": "frozen_before_model_iteration",
            "created_utc": v1._utc(),
            "policy": {
                "path": str((args.output_dir / "POLICY.json").resolve()),
                "sha256": policy_sha,
                "frozen_before_membership": True,
            },
            "repair": policy["repair"],
            "counts": {
                "source_inventory": len(inventory),
                "prior_excluded": len(excluded_fast),
                "globally_audited": len(completed),
                "eligible": len(eligible),
                "audit_excluded": len(completed) - len(eligible),
                "splits": split_counts,
                "source_groups": source_counts,
            },
            "assignment": assignment,
            "splits": {
                "development": development_manifest,
                "lockbox_manifest": v1._artifact(lockbox_manifest_path),
            },
            "artifacts": {
                "development_targets": v1._artifact(target_path),
                "exclusions": v1._artifact(exclusions_path),
                "leakage_report": v1._artifact(leakage_path),
                "audit_hash_cache": v1._artifact(cache_path),
                "lockbox_targets_encrypted": v1._artifact(encrypted_path),
                "lockbox_input_tree": {
                    "path": str(input_dir.resolve()),
                    "files": len(input_files),
                    "sha256": v1._tree_hash(input_files, args.output_dir),
                    "bytes": sum(path.stat().st_size for path in input_files),
                },
            },
            "target_invariants": {
                "global_exact_template_round_trip": True,
                "canonical_backward_steps": 0,
                "unique_rendered_identity": True,
                "raw_midi_temporal_order": True,
                "identity_crf_gold_coverage": 1.0,
                "planted_and_renderer_extra_distinguished": True,
            },
            "metric_gate": policy["gate"],
            "lockbox": {
                "sealed": True,
                "opened": False,
                "openings_allowed": 1,
                "target_content_public": False,
                "key_sha256": hashlib.sha256(key).hexdigest(),
            },
            "isolation": {
                "orn500_predictions_or_outcomes_read": False,
                "v1_lockbox_targets_read": False,
                "v1_lockbox_key_read": False,
                "original_mapper_v6_test_read": False,
                "production_weights_mutated": False,
                "paper_updated": False,
            },
        }
        manifest_path = args.output_dir / "release_manifest.json"
        v1._atomic_json(manifest_path, manifest)
        freeze_path = args.output_dir / "FREEZE.json"
        v1._atomic_json(
            freeze_path,
            {
                "schema_version": f"{SCHEMA_VERSION}-freeze",
                "frozen_utc": v1._utc(),
                "policy_sha256": policy_sha,
                "release_manifest": v1._artifact(manifest_path),
                "lockbox_manifest": v1._artifact(lockbox_manifest_path),
                "lockbox_targets_encrypted": v1._artifact(encrypted_path),
                "membership_frozen_before_training": True,
                "lockbox_opened": False,
            },
        )
        status = {
            "schema_version": f"{SCHEMA_VERSION}-status",
            "status": "frozen_ready_for_identity_crf_development",
            "release_manifest": v1._artifact(manifest_path),
            "freeze": v1._artifact(freeze_path),
            "policy_sha256": policy_sha,
            "split_counts": split_counts,
            "source_group_counts": source_counts,
            "eligible": len(eligible),
            "excluded": len(inventory) - len(eligible),
            "raw_lockbox_limitation": assignment[
                "raw_lockbox_limitation"
            ],
            "lockbox_opened": False,
            "production_weights_mutated": False,
            "paper_updated": False,
        }
        v1._atomic_json(args.output_dir / "STATUS.json", status)
        partial_path.unlink(missing_ok=True)
        print(json.dumps(status, indent=2), flush=True)
    finally:
        atexit.unregister(lease.__exit__)
        lease.__exit__(None, None, None)


def _resolve(repo: Path, value: Path) -> Path:
    return value if value.is_absolute() else repo / value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze-policy", "build"))
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--upstream-manifest", type=Path, default=DEFAULT_UPSTREAM
    )
    parser.add_argument("--v1-manifest", type=Path, default=DEFAULT_V1)
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
    args = parser.parse_args(argv)
    args.repo = args.repo.resolve()
    for name in (
        "output_dir",
        "upstream_manifest",
        "v1_manifest",
        "outputraw_targets",
        "outputraw_seal",
        "resource_status",
    ):
        setattr(args, name, _resolve(args.repo, getattr(args, name)))
    if args.phase == "freeze-policy":
        freeze_policy(args)
    else:
        if args.lockbox_key is None:
            raise ValueError("--lockbox-key outside the repository is required")
        args.lockbox_key = args.lockbox_key.resolve()
        try:
            args.lockbox_key.relative_to(args.repo)
        except ValueError:
            pass
        else:
            raise ValueError("Lockbox key must be outside the repository")
        build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
